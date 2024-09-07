import {
  Accordion,
  AccordionItem,
  AccordionButton,
  AccordionPanel,
  AccordionIcon,
  Box,
  List,
  ListItem,
  Text,
} from '@chakra-ui/react';
import { Link, useLocation } from 'react-router-dom';
import chapterList from '../data/sectionList';

const BookList = () => {
  const location = useLocation();

  const frontMatter = chapterList[0]
  const chapters = chapterList.slice(1, 13)
  const backMatter = chapterList.slice(-2)

  return (
    <Accordion allowToggle >
      {/* Front matter: Introduce chapter */}
      <List spacing={3} p='8px 16px'>
        <ListItem>
          <Link to={frontMatter.link}>
            <Text 
              fontSize='md' 
              _hover={{color: 'tomato'}}
              fontWeight={location.pathname === frontMatter.link ? 'bold' : 'normal'}
            >
              {frontMatter.name}
            </Text>
          </Link>
        </ListItem>
      </List>
      {/* 0 ~ 11 chapters */}
      {chapters.map((list) => (
        <AccordionItem key={list.id}>
          <AccordionButton>
            <Box flex='1' textAlign='left'>
              <Link to={list.link!}>
                <Text 
                  fontSize='md'
                  _hover={{color: 'tomato'}}
                  fontWeight={location.pathname === list.link ? 'bold' : 'normal'}
                >
                  {list.name}
                </Text>
              </Link>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel pb={4}>
            <List spacing={3}>
              {list.items?.map((item) => (
                <ListItem pl='22px' key={item.id}>
                  <Text 
                    fontSize='sm' 
                    _hover={{color: 'tomato'}}
                    color={location.hash === `#${item.link}` ? 'tomato' : 'gray.600'}
                  >
                    <a href={`#${item.link}`}>{item.name}</a>
                    </Text>
                </ListItem>
              ))}
            </List>
          </AccordionPanel>
        </AccordionItem>
      ))}
      {/*  Back Matter: Reference/About Me */}
        <List spacing={3} p='8px 16px'>
          {backMatter.map((m) => <ListItem key={m.id}>
            <Link to={m.link}>
              <Text 
                fontSize='md' 
                _hover={{color: 'tomato'}}
                fontWeight={location.pathname === m.link ? 'bold' : 'normal'}
              >
                {m.name}
              </Text>
            </Link>
          </ListItem>)}
        </List>
    </Accordion>
  );
};

export default BookList;
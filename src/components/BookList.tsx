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
  HStack,
} from '@chakra-ui/react';
import { Link } from 'react-router-dom';
import lists from '../data/chapterList'

const BookList = () => {
  const backMatter = [
    { id: 1, name: "Dedication", link: 'dedication' },
    { id: 2, name: "Additional Resources", link: 'additional-resources' },
    { id: 3, name: "Credits", link: 'credits' }
  ]
  return (
    <Accordion allowToggle >
      <List spacing={3} p='8px 16px'>
        <ListItem>
          <Link to={'introduction'}>
            <Text fontSize='md'>Introduction</Text>
          </Link>
        </ListItem>
      </List>
      {lists.map((list) => (
        <AccordionItem key={list.id}>
          <AccordionButton>
            <Box flex='1' textAlign='left'>
              <Link to={list.link!}>
                <HStack spacing={3}>
                  <Text>{list.id}</Text>
                  <Text fontSize='md'>{list.name}</Text>
                </HStack>
              </Link>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel pb={4}>
            <List spacing={3}>
              {list.items?.map((item) => (
                <ListItem pl='22px' key={item.id}>
                  <Text fontSize='sm'>{item.name}</Text>
                </ListItem>
              ))}
            </List>
          </AccordionPanel>
        </AccordionItem>
      ))}
        <List spacing={3} p='8px 16px'>
          {backMatter.map((m) => <ListItem key={m.id}>
            <Link to={m.link}>
              <Text fontSize='md'>{m.name}</Text>
            </Link>
          </ListItem>)}
        </List>
    </Accordion>
  );
};

export default BookList;
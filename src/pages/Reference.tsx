import { Container, Divider, Flex, Heading, Stack } from '@chakra-ui/react';
import ReferGrid from '../components/ReferGrid';
import reference from '../data/reference';
import FootLinks from '../components/FootLinks';

const Reference = () => {
  const [ 
    paper, statQuest, blueBrown, codeEmporium, others, webs ] = reference
  const footer = {
    l: "11. The Maths Behind (V)",
    r: "About Me",
    ll: "/the-maths-behind-five",
    rl: "/about-me"
  }

  
  return (
    <Container maxW='1200px' px='10px'>
      <Heading size='xl' color='gray.600' textAlign='center' py={3}>
        {'Reference'}
      </Heading>
      <ReferGrid section={paper}/>
      <Stack pt={3} pb={5} spacing={4}>
        <Heading as='b' fontSize='lg' color='gray.600' mt={3}>
          {'Videos'}
        </Heading>
        <ReferGrid section={statQuest}/>
        <ReferGrid section={blueBrown}/>
        <ReferGrid section={codeEmporium}/>
        <ReferGrid section={others}/>
      </Stack>
      <ReferGrid section={webs}/>
      <Flex align='center' h='80px'>
        <Divider variant='thick'/>
      </Flex>
      <FootLinks 
        l={footer.l} 
        ll={footer.ll} 
        r={footer.r} 
        rl={footer.rl}
      />
    </Container>
  )
}

export default Reference
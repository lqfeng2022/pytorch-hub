import { HStack, Icon, Text } from '@chakra-ui/react';
import { HiLanguage } from 'react-icons/hi2';

const NavBar = () => {
  return (
    <HStack
      justifyContent='space-between'
      padding='10px'
      margin={2}
      spacing={3}
    >
      <Text as='b' whiteSpace='nowrap' fontSize='2xl' color='tomato'>
        AI &nbsp;WITH &nbsp;PYTORCH
      </Text>
      <Icon boxSize='25px' as={HiLanguage} />
    </HStack>
  );
};

export default NavBar;
